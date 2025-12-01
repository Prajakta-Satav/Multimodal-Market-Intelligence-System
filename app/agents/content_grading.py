# app/agents/content_grading.py
from app.core.logging import logger
from app.models.state import AgentState, EvidenceChunk

def relevance(ch: EvidenceChunk, intent: str) -> float:
    """
    Compute a relevance score for a chunk based on:
    - Base similarity score
    - Intent alignment (fundamentals, prices, transcripts, presentations)
    - Recency and numeric content
    """
    base = ch.get("score", 0.0)

    # Intent alignment: boost chunks that match the query intent
    src_type = ch["meta"].get("type", "")
    if intent and src_type == intent:
        base += 0.15

    # Recency boost
    if "date" in ch["meta"]:
        base += 0.05

    # Numeric boost (financial data often has numbers)
    if any(c.isdigit() for c in ch["content"][:100]):
        base += 0.05

    return min(base, 1.0)

def run(state: AgentState) -> AgentState:
    chunks = state.get("retrieved_chunks", [])
    intent = state.get("intent", "general")

    # Score each chunk
    graded = [(relevance(ch, intent), ch) for ch in chunks]
    graded.sort(key=lambda x: x[0], reverse=True)

    # Filter: keep chunks above threshold, but ensure at least one survives
    filtered = [ch for score, ch in graded if score >= 0.4]
    if not filtered and graded:
        filtered = [graded[0][1]]

    # Synthesize context
    context = "\n\n".join([ch["content"] for ch in filtered[:6]])
    reasoning = f"Synthesized {len(filtered)} chunks; intent={intent}; applied relevance heuristic."

    # Risk proxy: lower risk if citations exist and multiple sources are used
    sources = {ch["source"] for ch in filtered}
    risk = 0.5
    if state.get("citations"):
        risk -= 0.2
    if len(sources) > 1:
        risk -= 0.1
    risk = max(risk, 0.0)

    # Update state
    state["consolidated_context"] = context
    state["reasoning_trace"] = reasoning
    state["hallucination_risk"] = risk
    state["graded_chunks"] = graded  # keep scores for debugging

    logger.info(f"[Grading] kept={len(filtered)} risk={risk:.2f} intent={intent}")
    return state
