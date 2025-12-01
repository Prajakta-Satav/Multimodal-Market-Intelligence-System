# app/models/state.py
from typing import TypedDict, List, Dict, Any, Optional

class EvidenceChunk(TypedDict):
    source: str           # chroma|minio|postgres
    id: str               # doc id, object key, row id
    content: str          # text snippet or structured serialization
    score: float          # relevance score
    meta: Dict[str, Any]  # extra metadata (timestamps, tickers, etc.)

class AgentState(TypedDict):
    # Input
    query: str
    # Query Understanding
    query_type: str
    entities: List[str]
    retrieval_strategy: str
    # Retrieval
    retrieved_chunks: List[EvidenceChunk]
    citations: List[str]
    # Grading & Reasoning
    consolidated_context: str
    reasoning_trace: str
    hallucination_risk: float
    # Answer
    answer: str
    output_format: str
    # Evaluation
    evaluation_score: float
    feedback_signal: Optional[str]  # "retry" | "accept"
