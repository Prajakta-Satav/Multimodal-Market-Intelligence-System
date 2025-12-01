# ============================================================================
# STEP-BY-STEP USAGE TEMPLATE (JSON Reference)
# ============================================================================

import json
import numpy as np

# STEP 1: Load your JSON file
json_file_path = r"D:\Multimodal-Market-Intelligence-System\ground_truth\qa_dataset.json"  # ← REPLACE WITH YOUR FILE NAME

with open(json_file_path, "r", encoding="utf-8") as f:
    eval_dataset = json.load(f)

print(f"Loaded {len(eval_dataset)} queries from {json_file_path}")

# STEP 2: Define evaluation metrics
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
    # Ideal: all golds first
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gold_chunks), k)))
    return dcg / ideal if ideal > 0 else 0.0

# STEP 3: Collect metrics for all queries
metrics = {
    'recall@5': [],
    'precision@5': [],
    'recall@10': [],
    'precision@10': [],
    'mrr': [],
    'ndcg@5': [],
    'ndcg@10': [],
}

for query in eval_dataset[:5]:  # just first 5
    gold_chunks = set(map(str, query.get('combined_chunk_ids', [])))
    retrieved = list(map(str, query.get('retrieved_chunk_ids', [])))

    print("Gold:", gold_chunks, "Retrieved:", retrieved)
    metrics['recall@5'].append(recall_at_k(gold_chunks, retrieved, 5))
    metrics['precision@5'].append(precision_at_k(gold_chunks, retrieved, 5))
    metrics['recall@10'].append(recall_at_k(gold_chunks, retrieved, 10))
    metrics['precision@10'].append(precision_at_k(gold_chunks, retrieved, 10))
    metrics['mrr'].append(mean_reciprocal_rank(gold_chunks, retrieved))
    metrics['ndcg@5'].append(ndcg_at_k(gold_chunks, retrieved, 5))
    metrics['ndcg@10'].append(ndcg_at_k(gold_chunks, retrieved, 10))

# STEP 4: Aggregate metrics
summary_metrics = {k: np.mean(v) for k, v in metrics.items()}
print("Summary Metrics:")
print(summary_metrics)

# STEP 5: (Optional) Export to CSV
import pandas as pd

df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("retrieval_evaluation.csv", index=False)
print("Metrics exported to retrieval_evaluation.csv")
# ============================================================================