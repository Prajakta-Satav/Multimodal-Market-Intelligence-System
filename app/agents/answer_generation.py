# app/agents/answer_generation.py
from app.core.logging import logger
from app.models.state import AgentState
from app.core.llm import call_llm


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    context = state.get("consolidated_context", "")
    entities = state.get("entities", [])

    prompt = f"""
    You are a financial research assistant.
    Question: {query}

    Context (retrieved evidence):
    {context}

    Entities: {', '.join(entities) if entities else 'None'}

    Task: Write a clear, concise, natural language answer to the question.
    - Use the context provided.
    - Mention key entities if relevant.
    - Keep the tone professional and informative.
    - Always produce at least one sentence, even if context is limited.
    - Do NOT include citation strings in the answer. Citations will be handled separately.
    """

    def build_fallback() -> str:
        return "\n".join([
            "Summary:",
            context[:800] if context else "(no context available)",
            "",
            "Key entities: " + (", ".join(entities) if entities else "None"),
        ])

    try:
        llm_answer = call_llm(prompt).strip()
        if not llm_answer or llm_answer.lower() == "(no answer generated)":
            logger.warning("[AnswerGeneration] Empty or invalid LLM output, using fallback")
            state["answer"] = build_fallback()
        else:
            state["answer"] = llm_answer
    except Exception as e:
        logger.error(f"[AnswerGeneration] LLM call failed: {e}")
        state["answer"] = build_fallback()

    # ✅ Format citations for user-facing response
    retrieved = state.get("retrieved_chunks", [])
    state["citations"] = [format_citation(ch) for ch in retrieved]

    state["output_format"] = "markdown"
    logger.info(f"[AnswerGeneration] produced answer with {len(state['citations'])} citations")
    return state


def format_citation(chunk) -> str:
    src = chunk.get("source", "")
    meta = chunk.get("meta", {})

    # --- Postgres sources ---
    if src.startswith("postgres:stock_prices"):
        return f"Stock Prices (Ticker={meta.get('ticker')}, Date={meta.get('date')})"
    elif src.startswith("postgres:stock_fundamentals"):
        return f"Fundamentals (Ticker={meta.get('ticker')}, Year={meta.get('year')})"
    elif src.startswith("postgres:transcripts"):
        quarter = meta.get("quarter")
        year = meta.get("year")
        ticker = meta.get("ticker")
        return f"Transcript ({ticker} Q{quarter} {year})"
    elif src.startswith("postgres:presentations"):
        return f"Presentation ({meta.get('ticker')} {meta.get('year')})"

    # --- MinIO / Chroma sources ---
    elif src.startswith("minio:") or src.startswith("chroma:"):
        # Try to show filename or document title
        filename = meta.get("filename") or meta.get("doc_name") or meta.get("id")
        return f"Document: {filename}"

    # --- Fallback ---
    else:
        return src
