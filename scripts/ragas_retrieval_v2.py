# ============================================================================
# Enhanced Retrieval Evaluation (Semantic + LLM RAGAS-style)
# ============================================================================
import os
import json
import numpy as np
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embeddings import EmbeddingService   # your embedding service
import google.generativeai as genai

# -------------------------------
# Config
# -------------------------------
PERSIST_DIR = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\data\chroma"
QA_FILE = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\ground_truth\qa_dataset.json"
COLLECTION_NAME = "mmi_index"
EMBEDDING_MODEL_NAME = "e5-large"
N_RESULTS = 10
SIM_THRESHOLD = 0.75
K_VALUES = [1, 3, 5, 7, 10]

# Gemini setup
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = ""
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    GEMINI_MODEL_NAME = "gemini-2.5-flash"
else:
    GEMINI_MODEL_NAME = None

# -------------------------------
# Standard IR metrics
# -------------------------------
def mean_reciprocal_rank(relevant: set, retrieved: list) -> float:
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevant: set, retrieved: list, k: int) -> float:
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        rel = 1 if chunk_id in relevant else 0
        dcg += rel / np.log2(i + 2)
    ideal_rels = [1] * min(len(relevant), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

# -------------------------------
# Semantic metrics
# -------------------------------
def semantic_precision_recall(
    embedder, query_text, retrieved_texts, gold_texts, threshold=0.75
):
    """Compute semantic precision and recall using cosine similarity."""
    if not retrieved_texts or not gold_texts:
        return 0.0, 0.0

    # Embed everything
    query_emb = embedder.embed_query(query_text)
    retr_emb = [embedder.embed_query(t) for t in retrieved_texts]
    gold_emb = [embedder.embed_query(t) for t in gold_texts]

    # Similarity matrix
    sim_matrix = cosine_similarity(retr_emb, gold_emb)

    # Precision: fraction of retrieved chunks that are relevant
    relevant_retrieved = sum(
        1 for row in sim_matrix if np.max(row) >= threshold
    )
    precision = relevant_retrieved / len(retrieved_texts)

    # Recall: fraction of gold chunks covered
    relevant_gold = sum(
        1 for col in sim_matrix.T if np.max(col) >= threshold
    )
    recall = relevant_gold / len(gold_texts)

    return precision, recall

# -------------------------------
# LLM-based RAGAS scoring
# -------------------------------
LLM_PROMPT = """You are evaluating retrieval quality for a RAG system.

Query: {query}

Retrieved Chunks:
{retrieved_chunks}

Gold Chunks:
{gold_chunks}

Rate the following metrics between 0.0 and 1.0 with one-sentence reasoning:
Faithfulness: <score> - <reason>
Answer Relevancy: <score> - <reason>
Context Precision: <score> - <reason>
Context Recall: <score> - <reason>
"""

def llm_ragas(query, retrieved_texts, gold_texts):
    if GEMINI_MODEL_NAME is None:
        return "LLM scoring skipped (no GEMINI_API_KEY)."

    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    retrieved_fmt = "\n".join(
        f"- Retrieved#{i+1}: {t[:300]}..." for i, t in enumerate(retrieved_texts)
    )
    gold_fmt = "\n".join(
        f"- Gold#{i+1}: {t[:300]}..." for i, t in enumerate(gold_texts)
    )

    prompt = LLM_PROMPT.format(
        query=query, retrieved_chunks=retrieved_fmt, gold_chunks=gold_fmt
    )

    resp = model.generate_content(prompt)
    return getattr(resp, "text", "").strip()

# -------------------------------
# Main
# -------------------------------
def main():
    # Connect to Chroma
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Load QA dataset
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)

    embedder = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)

    query_text = input("Enter your query: ").strip()
    query_emb = embedder.embed_query(query_text)

    results = collection.query(
    query_embeddings=[query_emb],
    n_results=10,
    include=["documents", "metadatas", "distances"]
)

    retrieved_ids = results["ids"][0]          # IDs are always included
    retrieved_docs = results["documents"][0]   # text chunks
    retrieved_scores = results["distances"][0] # similarity scores
    retrieved_meta = results["metadatas"][0]   # optional metadata

    # Match query in QA dataset
    matched = next(
        (q for q in qa_dataset if q["question"].strip().lower() == query_text.lower()),
        None,
    )
    if not matched:
        print("⚠️ Query not found in QA dataset.")
        return

    gold_ids = [str(x) for x in matched.get("combined_chunk_ids", [])]
    gold_texts = matched.get("combined_chunks_text", [])
    if not gold_texts:
        fetched = collection.get(ids=gold_ids)
        gold_texts = fetched.get("documents", [])

    gold_set = set(gold_ids)

    # Standard metrics
    metrics = {}
    for k in K_VALUES:
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_set, retrieved_ids, k)
    metrics["mrr"] = mean_reciprocal_rank(gold_set, retrieved_ids)

    # Semantic metrics
    sem_precision, sem_recall = semantic_precision_recall(
        embedder, query_text, retrieved_docs, gold_texts, threshold=SIM_THRESHOLD
    )

    # LLM scoring
    llm_text = llm_ragas(query_text, retrieved_docs, gold_texts)

    # Report
    print("\n📊 Retrieval Evaluation Report")
    print("Standard IR metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\nSemantic Precision: {sem_precision:.4f}")
    print(f"Semantic Recall: {sem_recall:.4f}")

    print("\n🧠 LLM-based RAGAS-style assessment:")
    print(llm_text)

if __name__ == "__main__":
    main()
