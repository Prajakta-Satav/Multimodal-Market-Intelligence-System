# app/services/multi_index_retrieval.py

from typing import List, Dict, Optional
from app.services.embeddings import EmbeddingService
from app.services.chroma_client import ChromaService
from app.core.logging import logger

embedder = EmbeddingService(model_name="e5-large")
chroma = ChromaService(persist_dir="data/chroma")

# ---- utilities ---------------------------------------------------------

TYPE_ALIAS = {
    "presentation": "ppt",  # user-facing alias -> stored type
    "ppt": "ppt",
    "earnings_release": "earnings_release",
    "transcript": "transcript",
    "fundamentals": "fundamentals",
    "prices": "prices",
}

def _resolve_type(t: str) -> str:
    return TYPE_ALIAS.get(t, t)

def _safe_extract(res: Dict, key: str, default=None):
    return res.get(key, default)

def _flatten_chroma_result(res: Dict, type_name: str) -> List[Dict]:
    """Flatten chroma query/get response into list of uniform dicts."""
    docs = _safe_extract(res, "documents", [[]])
    metas = _safe_extract(res, "metadatas", [[]])
    ids = _safe_extract(res, "ids", [[]])
    dists = _safe_extract(res, "distances", [[]])
    
    # Handle both query() and get() response formats
    if docs and isinstance(docs[0], list):
        # query() returns nested lists
        docs = docs[0] if docs else []
        metas = metas[0] if metas else []
        ids = ids[0] if ids else []
        dists = dists[0] if dists else []
    
    items = []
    for i in range(len(docs)):
        items.append({
            "doc": docs[i],
            "meta": metas[i] if i < len(metas) else {},
            "id": ids[i] if i < len(ids) else f"unknown_{i}",
            "dist": dists[i] if i < len(dists) else 1.0,
            "type": type_name
        })
    
    return items

# ---- retrieval strategies ----------------------------------------------

def semantic_search(query: str, types: Optional[List[str]] = None, n_results: int = 10) -> List[Dict]:
    """
    Semantic search across all or filtered types.
    """
    q_emb = embedder.embed([query], is_query=True)
    
    if types:
        resolved_types = [_resolve_type(t) for t in types]
        where_clause = {"type": {"$in": resolved_types}}
        res = chroma.query(query_embeddings=q_emb, n_results=n_results, where=where_clause)
    else:
        res = chroma.query(query_embeddings=q_emb, n_results=n_results)
    
    return _flatten_chroma_result(res, "mixed")

def filter_by_metadata(types: Optional[List[str]] = None, n_results: int = 10) -> List[Dict]:
    """
    Direct metadata filtering without semantic search.
    """
    if not types:
        logger.warning("[MultiIndex] filter_by_metadata called without types, returning empty")
        return []
    
    resolved_types = [_resolve_type(t) for t in types]
    where_clause = {"type": {"$in": resolved_types}}
    
    res = chroma.get(where=where_clause, limit=n_results)
    return _flatten_chroma_result(res, "filtered")

def hybrid_retrieval(
    query: str,
    types: Optional[List[str]] = None,
    n_results: int = 10,
    rerank: bool = True
) -> List[Dict]:
    """
    Combine semantic search with metadata filtering.
    """
    items = semantic_search(query, types, n_results * 2)  # get more candidates
    
    if rerank:
        # Simple reranking: prioritize by type order if provided
        if types:
            type_priority = {_resolve_type(t): i for i, t in enumerate(types)}
            
            def sort_key(item):
                item_type = item["meta"].get("type", "unknown")
                priority = type_priority.get(item_type, len(types))
                return (priority, item["dist"])
            
            items.sort(key=sort_key)
    
    return items[:n_results]

# ---- multi-index query -------------------------------------------------

def multi_index_query(
    query: str,
    n_per_index: int = 3,
    final_n: int = 3,
    strategy: str = "hybrid"
) -> List[Dict]:
    """
    Query multiple index types and merge results.
    
    Args:
        query: User query
        n_per_index: How many results to fetch per type
        final_n: Final number of results after reranking
        strategy: "semantic", "filter", or "hybrid"
    """
    index_types = ["transcript", "ppt", "fundamentals", "prices"]
    all_items = []
    
    for idx_type in index_types:
        if strategy == "semantic":
            items = semantic_search(query, types=[idx_type], n_results=n_per_index)
        elif strategy == "filter":
            items = filter_by_metadata(types=[idx_type], n_results=n_per_index)
        else:  # hybrid
            items = hybrid_retrieval(query, types=[idx_type], n_results=n_per_index, rerank=False)
        
        all_items.extend(items)
    
    # Global reranking by distance
    all_items.sort(key=lambda x: x["dist"])
    
    logger.info(f"[MultiIndex] Retrieved {len(all_items)} items, returning top {final_n}")
    return all_items[:final_n]
