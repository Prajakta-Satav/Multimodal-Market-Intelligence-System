# ============================================================================
# RAGAS-Style Retrieval Evaluation (ChromaDB + QA JSON + E5-large + Gemini)
# Retrieval-level scoring: Faithfulness, Answer Relevancy, Context Precision, Context Recall
# ============================================================================
import os
import json
import numpy as np
import chromadb
from typing import List, Dict, Any, Tuple
from app.services.embeddings import EmbeddingService  # your embedding service
from sklearn.metrics.pairwise import cosine_similarity

# Optional: Gemini (LLM-based scoring) - textual outputs
# pip install google-generativeai
import google.generativeai as genai

# -------------------------------
# Config
# -------------------------------
PERSIST_DIR = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\data\chroma"
QA_FILE = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\ground_truth\qa_dataset.json"
COLLECTION_NAME = "mmi_index"
EMBEDDING_MODEL_NAME = "e5-large"
N_RESULTS = 10
K_VALUES = [1, 3, 5, 7, 10]

import os

# Gemini setup (expects environment variable GEMINI_API_KEY)
# GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = "AIzaSyBYG0hlhhiSGPFD1ZmEyKGkedUmdaqmoNg"

print("Gemini key loaded:", os.getenv("GEMINI_API_KEY") is not None)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-2.5-flash"  # lightweight, fast; change if needed
else:
    GEMINI_MODEL_NAME = None  # LLM scoring will be skipped if no key present

# -------------------------------
# Standard retrieval metrics
# -------------------------------
def precision_at_k(gold_chunks: set, retrieved: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & gold_chunks) / k

def recall_at_k(gold_chunks: set, retrieved: List[str], k: int) -> float:
    if not gold_chunks:
        return 0.0
    return len(set(retrieved[:k]) & gold_chunks) / len(gold_chunks)

def mean_reciprocal_rank(gold_chunks: set, retrieved: List[str]) -> float:
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in gold_chunks:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(gold_chunks: set, retrieved: List[str], k: int) -> float:
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        rel = 1 if chunk_id in gold_chunks else 0
        dcg += rel / np.log2(i + 2)
    ideal_rels = [1] * min(len(gold_chunks), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

# -------------------------------
# Helpers: Chroma interactions
# -------------------------------
def get_gold_chunk_texts(
    collection: chromadb.api.models.Collection.Collection,
    gold_ids: List[str]
) -> List[str]:
    """Fetch texts for gold chunk IDs from Chroma, if not present in QA file."""
    if not gold_ids:
        return []
    # Chroma get() can retrieve by IDs; returns dict with 'documents'
    fetched = collection.get(ids=gold_ids)
    docs = fetched.get("documents", [])
    return docs or []

def ensure_list_strings(xs: Any) -> List[str]:
    if not xs:
        return []
    return [str(x) for x in xs]

# -------------------------------
# LLM-based scoring (Gemini)
# -------------------------------
LLM_PROMPT_TEMPLATE = """You are evaluating retrieval quality for a RAG system at retrieval-level only.

Inputs:
- Query: "{query}"
- Retrieved Chunks (top-N): 
{retrieved_chunks}
- Gold Chunks (expected context): 
{gold_chunks}

Task:
Rate the following four metrics strictly between 0.0 and 1.0 and explain briefly:
1) Faithfulness: Do the retrieved chunks align with and not contradict the gold context? (0–1)
2) Answer Relevancy: Are the retrieved chunks responsive to the query? (0–1)
3) Context Precision: Among the retrieved chunks, how much is actually useful for answering the query? (0–1)
4) Context Recall: Do the retrieved chunks collectively contain all necessary information found in the gold context? (0–1)

Guidance:
- Consider semantic alignment, topicality, and coverage of facts.
- If retrieved content overlaps well with gold but includes some irrelevant noise, lower precision.
- If retrieved content misses key gold facts, lower recall.
- Provide concise reasoning for each score.

Output:
Return a short textual assessment with each metric on its own line in the format:
Faithfulness: <score> - <one sentence reason>
Answer Relevancy: <score> - <one sentence reason>
Context Precision: <score> - <one sentence reason>
Context Recall: <score> - <one sentence reason>
"""

def format_chunks_for_prompt(chunks: List[str], label: str, max_chars: int = 4000) -> str:
    """Concise chunk formatting to avoid excessive token usage."""
    lines = []
    total = 0
    for i, c in enumerate(chunks):
        snippet = c.strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600] + "..."
        entry = f"- {label}#{i+1}: {snippet}"
        if total + len(entry) > max_chars:
            lines.append(f"... (truncated {len(chunks) - i} more)")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines) if lines else "- (none)"

def llm_score_retrieval(query: str, retrieved_texts: List[str], gold_texts: List[str]) -> str:
    """Call Gemini to produce textual scores and brief reasoning."""
    if GEMINI_MODEL_NAME is None:
        return ("LLM scoring skipped (GEMINI_API_KEY not set).\n"
                "Faithfulness: N/A\nAnswer Relevancy: N/A\nContext Precision: N/A\nContext Recall: N/A")

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    retrieved_fmt = format_chunks_for_prompt(retrieved_texts, label="retrieved")
    gold_fmt = format_chunks_for_prompt(gold_texts, label="gold")

    prompt = LLM_PROMPT_TEMPLATE.format(
        query=query,
        retrieved_chunks=retrieved_fmt,
        gold_chunks=gold_fmt
    )
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "").strip()
    return text or "LLM returned no text."

# -------------------------------
# Relative semantic score (embedding-based)
# -------------------------------
def relative_semantic_score(
    embedder: EmbeddingService,
    retrieved_texts: List[str],
    gold_texts: List[str]
) -> float:
    """Average cosine similarity between retrieved and gold chunks."""
    if not retrieved_texts or not gold_texts:
        return 0.0
    retr_emb = [embedder.embed_query(t) for t in retrieved_texts]
    gold_emb = [embedder.embed_query(t) for t in gold_texts]
    sim_matrix = cosine_similarity(retr_emb, gold_emb)
    return float(np.mean(sim_matrix))

# -------------------------------
# Main
# -------------------------------
def main():
    # Step 1: Connect to Chroma
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    print(client.list_collections())
    collection = client.get_collection(name=COLLECTION_NAME)

    # Step 2: Load QA dataset
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    print(f"✅ Loaded {len(qa_dataset)} queries from {QA_FILE}")

    # Step 3: Initialize embedding service
    embedder = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)

    # Step 4: Interactive query input
    query_text = input("Enter your query: ").strip()

    # Step 5: Generate query embedding and retrieve from Chroma
    query_embedding = embedder.embed_query(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS,
        include=["documents", "metadatas", "distances"]
    )
    retrieved_ids = results["ids"][0]
    retrieved_docs = results["documents"][0]

    print("\n📌 Top retrieved chunk IDs:")
    print(retrieved_ids)

    # Step 6: Match query in QA dataset
    matched = None
    for q in qa_dataset:
        if q["question"].strip().lower() == query_text.strip().lower():
            matched = q
            break

    if not matched:
        print("\n⚠️ Query not found in QA dataset, skipping evaluation.")
        return

    gold_ids = ensure_list_strings(matched.get("combined_chunk_ids", []))
    gold_texts_in_file = matched.get("combined_chunks_text", [])
    gold_texts_in_file = gold_texts_in_file if isinstance(gold_texts_in_file, list) else []
    gold_texts = gold_texts_in_file

    if not gold_texts:
        # Fetch gold texts from Chroma using IDs
        gold_texts = get_gold_chunk_texts(collection, gold_ids)

    gold_set = set(gold_ids)

    print("\n🎯 Gold chunk IDs:")
    print(gold_set)

    # Step 7: Overlap evaluation
    overlap = set(retrieved_ids) & gold_set
    if overlap:
        print("\n✅ Good Result: Retrieved includes gold answer context.")
        print(f"Matched gold chunks: {overlap}")
    else:
        print("\nℹ️ Relative Result: Retrieved did not include the gold answer context.")

    # Step 8: Standard metrics
    metrics = {}
    for k in K_VALUES:
        metrics[f"precision@{k}"] = precision_at_k(gold_set, retrieved_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(gold_set, retrieved_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_set, retrieved_ids, k)
    metrics["mrr"] = mean_reciprocal_rank(gold_set, retrieved_ids)

    print("\n📊 Retrieval Metrics (standard):")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Step 9: Relative semantic score (embedding-based)
    rel_score = relative_semantic_score(embedder, retrieved_docs, gold_texts)
    print(f"\n🔎 Relative semantic similarity (retrieved vs gold): {rel_score:.4f}")

    # Step 10: LLM-based RAGAS-style retrieval scoring (textual)
    print("\n🧠 LLM-based RAGAS-style retrieval assessment:")
    llm_text = llm_score_retrieval(
        query=query_text,
        retrieved_texts=retrieved_docs,
        gold_texts=gold_texts
    )
    print(llm_text)

    # Optional: compact summary
    print("\n— Summary —")
    print(f"Overlap with gold: {'Yes' if overlap else 'No'}")
    print(f"Relative semantic similarity: {rel_score:.4f}")
    print("See above for LLM textual scores (Faithfulness, Answer Relevancy, Context Precision, Context Recall).")

if __name__ == "__main__":
    main()
