# app/core/errors.py
class AgentError(Exception):
    """Base error for agent failures."""

class RetrievalError(AgentError):
    """Raised when retrieval fails or returns insufficient content."""

class ValidationError(AgentError):
    """Raised when inputs/outputs violate schema or constraints."""
