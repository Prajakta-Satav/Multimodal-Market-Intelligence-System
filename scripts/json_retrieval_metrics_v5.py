# ============================================================================
# Retrieval Evaluation Script (ChromaDB + QA JSON + E5-large + Metrics + Relative Score)
# ============================================================================
import json
import numpy as np
import chromadb
from app.services.embeddings import EmbeddingService   # <-- use your embedding service
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# STEP 1: Connect to Chroma
# -------------------------------
persist_dir = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\data\chroma"
client = chromadb.PersistentClient(path=persist_dir)

print(client.list_collections())

collection = client.get_collection(name="mmi_index")

# -------------------------------
# STEP 2: Load QA dataset (gold labels)
# -------------------------------
qa_file = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\ground_truth\qa_dataset.json"

with open(qa_file, "r", encoding="utf-8") as f:
    qa_dataset = json.load(f)

print(f"✅ Loaded {len(qa_dataset)} queries from {qa_file}")

# -------------------------------
# STEP 3: Define metrics
# -------------------------------
def precision_at_k(gold_chunks, retrieved, k):
    if k == 0: return 0.0
    return len(set(retrieved[:k]) & set(gold_chunks)) / k

def recall_at_k(gold_chunks, retrieved, k):
    if not gold_chunks: return 0.0
    return len(set(retrieved[:k]) & set(gold_chunks)) / len(gold_chunks)

def mean_reciprocal_rank(gold_chunks, retrieved):
    for i, chunk_id in enumerate(retrieved):
        if chunk_id in gold_chunks:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(gold_chunks, retrieved, k):
    dcg = 0.0
    for i, chunk_id in enumerate(retrieved[:k]):
        rel = 1 if chunk_id in gold_chunks else 0
        dcg += rel / np.log2(i + 2)
    ideal_rels = [1] * min(len(gold_chunks), k)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

# -------------------------------
# STEP 4: Initialize embedding service
# -------------------------------
embedder = EmbeddingService(model_name="e5-large")

# -------------------------------
# STEP 5: Interactive query input
# -------------------------------
query_text = input("Enter your query: ").strip()

query_embedding = embedder.embed_query(query_text)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

retrieved_ids = results["ids"][0]
retrieved_docs = results["documents"][0]  # actual text chunks

print("\n📌 Top retrieved chunks:")
print(retrieved_ids)

# -------------------------------
# STEP 6: Match query in QA dataset
# -------------------------------
matched = None
for q in qa_dataset:
    if q["question"].strip().lower() == query_text.strip().lower():
        matched = q
        break

if not matched:
    print("\n⚠️ Query not found in QA dataset, skipping evaluation.")
else:
    gold_chunks = set(map(str, matched.get("combined_chunk_ids", [])))

    print("\n🎯 Gold chunks for this query:")
    print(gold_chunks)

    # -------------------------------
    # STEP 7: Evaluate retrieval outcome
    # -------------------------------
    overlap = set(retrieved_ids) & gold_chunks

    if overlap:
        print("\n✅ Good Result: Retrieved chunks include the gold answer context.")
        print(f"Matched gold chunks: {overlap}")
    else:
        print("\nℹ️ Relative Result: Retrieved chunks did not include the gold answer context.")

        # Compute relative score (semantic similarity)
        gold_embeddings = [embedder.embed_query(doc) for doc in matched.get("combined_chunks_text", [])]
        retrieved_embeddings = [embedder.embed_query(doc) for doc in retrieved_docs]

        if gold_embeddings and retrieved_embeddings:
            sim_matrix = cosine_similarity(retrieved_embeddings, gold_embeddings)
            relative_score = float(np.mean(sim_matrix))
            print(f"🔎 Relative semantic similarity score: {relative_score:.4f}")

    # -------------------------------
    # STEP 8: Numerical metrics
    # -------------------------------
    k_values = [1, 3, 5, 7, 10]
    metrics = {}

    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(gold_chunks, retrieved_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(gold_chunks, retrieved_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_chunks, retrieved_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(gold_chunks, retrieved_ids)

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
