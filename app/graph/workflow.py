# app/graph/workflow.py
from langgraph.graph import StateGraph
from app.models.state import AgentState
from app.agents import (
    query_understanding,
    rag_retrieval,
    content_grading,
    answer_generation,
    # self_evaluation,   # <-- keep aside for now
)

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("query_understanding", query_understanding.run)
workflow.add_node("rag_retrieval", rag_retrieval.run)
workflow.add_node("content_grading", content_grading.run)
workflow.add_node("answer_generation", answer_generation.run)
# workflow.add_node("self_evaluation", self_evaluation.run)  # <-- commented out

# Define edges (order of execution)
workflow.add_edge("query_understanding", "rag_retrieval")
workflow.add_edge("rag_retrieval", "content_grading")
workflow.add_edge("content_grading", "answer_generation")
# workflow.add_edge("answer_generation", "self_evaluation")  # <-- commented out

# Set entry point and terminal node
workflow.set_entry_point("query_understanding")
workflow.set_finish_point("answer_generation")  # <-- finish at answer_generation

# Compile the graph
workflow = workflow.compile()
