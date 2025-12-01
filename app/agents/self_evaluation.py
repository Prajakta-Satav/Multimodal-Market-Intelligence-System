# app/agents/self_evaluation.py
from app.core.logging import logger
from app.core.config import settings
from app.models.state import AgentState

def run(state: AgentState) -> AgentState:
    citations = state.get("citations", [])
    risk = state.get("hallucination_risk", 0.5)
    graded_chunks = state.get("graded_chunks", [])
    reasoning = state.get("reasoning_trace", "")

    # Citation factor: more citations = higher confidence, capped
    citation_factor = 0.6 + 0.1 * min(len(citations), 3)

    # Diversity factor: boost if multiple sources are represented
    try:
        sources = {ch["source"] for ch in graded_chunks} if graded_chunks else set()
    except Exception:
        sources = set()

    diversity_factor = 1.0 if len(sources) > 1 else 0.9

    # Coverage factor: boost if we kept more than 3 chunks
    coverage_factor = 1.0 if len(graded_chunks) >= 3 else 0.85

    # Final confidence score
    confidence = (1 - risk) * citation_factor * diversity_factor * coverage_factor
    state["evaluation_score"] = round(confidence, 3)

    # Feedback signal
    state["feedback_signal"] = "retry" if confidence < settings.EVAL_CONFIDENCE_THRESHOLD else "accept"

    # Trace logging
    state.setdefault("trace", []).append({
        "agent": "self_evaluation",
        "score": state["evaluation_score"],
        "signal": state["feedback_signal"],
        "reasoning": reasoning
    })

    # ✅ Preserve the answer so it survives to the final state
    if "answer" not in state or not state["answer"]:
        # If answer_generation already set it, keep it
        # If not, set a placeholder
        state["answer"] = state.get("answer", "(no answer available)")

    logger.info(
        f"[Eval] score={state['evaluation_score']} "
        f"signal={state['feedback_signal']} "
        f"citations={len(citations)} sources={len(sources)} risk={risk}"
    )
    return state
