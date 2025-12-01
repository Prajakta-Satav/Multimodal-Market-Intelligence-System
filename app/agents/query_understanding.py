# app/agents/query_understanding.py
import re
from app.core.logging import logger
from app.models.state import AgentState

FINANCE_KEYWORDS = {
    "fundamentals": ["balance sheet", "income statement", "fundamentals", "valuation"],
    "prices": ["price", "stock", "market cap", "share"],
    "transcripts": ["earnings call", "transcript", "Q&A"],
    "presentations": ["slides", "presentation", "deck"],
}

def extract_entities(query: str):
    # Very naive entity extractor: look for capitalized words or tickers like AAPL, MSFT
    tickers = re.findall(r"\b[A-Z]{2,5}\b", query)
    companies = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)
    return list(set(tickers + companies))

def classify_intent(query: str):
    q_lower = query.lower()
    for intent, keywords in FINANCE_KEYWORDS.items():
        if any(k in q_lower for k in keywords):
            return intent
    return "general"

def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    entities = extract_entities(query)
    intent = classify_intent(query)

    # Decide retrieval strategy: semantic for general, keyword for specific tickers
    strategy = "semantic" if intent == "general" else "hybrid"

    state["entities"] = entities
    state["intent"] = intent
    state["retrieval_strategy"] = strategy
    state["parsed_query"] = query.strip()

    logger.info(f"[QueryUnderstanding] intent={intent}, strategy={strategy}, entities={entities}")
    return state
