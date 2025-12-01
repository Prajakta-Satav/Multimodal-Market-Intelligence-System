# ============================================================================
# Standalone Retrieval Evaluation Script (ChromaDB + QA JSON + E5-large)
# ============================================================================
import json
import numpy as np
import chromadb
from app.services.embeddings import EmbeddingService   # <-- use your embedding service

# -------------------------------
# STEP 1: Connect to Chroma
# -------------------------------
persist_dir = r"D:\Workspace\POC\multimodal_financial_agent\Multimodal-Market-Intelligence-System\data\chroma"
client = chromadb.PersistentClient(path=persist_dir)

print(client.list_collections())

# Adjust collection name to match what you used when inserting
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
def recall_at_k(gold_chunks, retrieved, k):
    return len(set(retrieved[:k]) & set(gold_chunks)) / len(gold_chunks) if gold_chunks else 0.0

def precision_at_k(gold_chunks, retrieved, k):
    return len(set(retrieved[:k]) & set(gold_chunks)) / k

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
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gold_chunks), k)))
    return dcg / ideal if ideal > 0 else 0.0

# -------------------------------
# STEP 4: Initialize embedding service
# -------------------------------
embedder = EmbeddingService(model_name="e5-large")

# -------------------------------
# STEP 5: Interactive query input
# -------------------------------
query_text = input("Enter your query: ").strip()

# Generate query embedding with E5-large
query_embedding = embedder.embed_query(query_text)

# Run retrieval from Chroma using embeddings (not query_texts!)
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

retrieved_ids = results["ids"][0]  # list of retrieved chunk IDs

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

    # -------------------------------
    # STEP 7: Evaluate metrics
    # -------------------------------
    k_values = [1, 3, 5, 7, 10]
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(gold_chunks, retrieved_ids, k)
        metrics[f"precision@{k}"] = precision_at_k(gold_chunks, retrieved_ids, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(gold_chunks, retrieved_ids, k)

    metrics["mrr"] = mean_reciprocal_rank(gold_chunks, retrieved_ids)

    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
