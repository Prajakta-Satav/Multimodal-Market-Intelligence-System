"""Simple RAG test harness.

Usage:
  python .\scripts\test_rag.py --query "What did Amazon say about AWS?"

This loads the query understanding, retrieval, answer generation and evaluation agents
and runs them sequentially, printing the final answer and metadata.
"""
import argparse
from app.models.state import AgentState
from app.agents.query_understanding import run as uq_run
from app.agents.rag_retrieval import run as rag_run
from app.agents.answer_generation import run as ag_run
from app.agents.self_evaluation import run as eval_run
from app.core.logging import logger


def run_rag(query: str):
    state: AgentState = {
        "query": query,
        "query_type": "",
        "entities": [],
        "retrieval_strategy": "",
        "retrieved_chunks": [],
        "citations": [],
        "consolidated_context": "",
        "reasoning_trace": "",
        "hallucination_risk": 0.5,
        "final_answer": "",
        "output_format": "",
        "evaluation_score": 0.0,
        "feedback_signal": None
    }

    # Query understanding
    state = uq_run(state)

    # Retrieval
    state = rag_run(state)

    # Answer generation
    state = ag_run(state)

    # Self evaluation
    state = eval_run(state)

    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', '-q', required=True, help='Query to run through RAG')
    args = p.parse_args()

    result = run_rag(args.query)

    print('\n=== FINAL ANSWER ===')
    print(result.get('final_answer', ''))
    print('\n=== METADATA ===')
    print('Citations:', result.get('citations'))
    print('Retrieved chunks:', len(result.get('retrieved_chunks', [])))
    print('Eval score:', result.get('evaluation_score'))


if __name__ == '__main__':
    main()
