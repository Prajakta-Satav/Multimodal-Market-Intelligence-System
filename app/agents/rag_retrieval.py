# app/agents/rag_retrieval.py
from typing import List
from app.core.logging import logger
from app.core.errors import RetrievalError
from app.models.state import AgentState, EvidenceChunk
from app.services.multi_index_retrieval import multi_index_query

def run(state: AgentState) -> AgentState:
    # Use parsed_query if available, otherwise fall back to raw query
    query = state.get("parsed_query") or state.get("query")
    if not query:
        state["feedback_signal"] = "retry"
        raise RetrievalError("No query provided to rag_retrieval")

    strategy = state.get("retrieval_strategy", "semantic")
    logger.info(f"[RAG] strategy={strategy} query='{query}'")

    # Call multi_index_query without unsupported 'mode' argument
    results = multi_index_query(query, n_per_index=3, final_n=6)

    chunks: List[EvidenceChunk] = []
    for r in results:
        meta = r.get("meta", {})
        src = meta.get("source", meta.get("table", "chroma"))
        cid = r["id"]
        score = 1.0 - float(r["dist"]) if "dist" in r else r.get("score", 0.0)
        content = r["doc"]
        chunks.append({
            "source": src,
            "id": cid,
            "content": content,
            "score": score,
            "meta": meta
        })

    if not chunks:
        state["feedback_signal"] = "retry"
        raise RetrievalError("No evidence chunks retrieved")

    # --- Citations ---
    citations = []
    for ch in chunks[:3]:
        src = ch["source"]
        meta = ch["meta"]

        if src == "minio" and "object_key" in meta:
            citations.append(f"minio://{meta['object_key']}")
        elif src == "postgres":
            citations.append(f"postgres://{meta.get('table')}/{ch['id']}")
        elif src == "chroma":
            citations.append(f"chroma://{ch['id']}")
        else:
            # generic fallback for unknown sources
            citations.append(f"{src}://{ch['id']}")

    state["retrieved_chunks"] = sorted(chunks, key=lambda x: x["score"], reverse=True)
    state["citations"] = citations

    logger.info(f"[RAG] retrieved={len(state['retrieved_chunks'])} citations={len(citations)}")
    return state
