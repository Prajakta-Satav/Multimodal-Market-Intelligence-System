# ============================================================================
# Retrieval Evaluation Script (ChromaDB + QA JSON + E5-large)
# ============================================================================
import json
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
# STEP 3: Initialize embedding service
# -------------------------------
embedder = EmbeddingService(model_name="e5-large")

# -------------------------------
# STEP 4: Interactive query input
# -------------------------------
query_text = input("Enter your query: ").strip()

# Generate query embedding with E5-large
query_embedding = embedder.embed_query(query_text)

# Run retrieval from Chroma using embeddings
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10
)

retrieved_ids = results["ids"][0]  # list of retrieved chunk IDs

print("\n📌 Top retrieved chunks:")
print(retrieved_ids)

# -------------------------------
# STEP 5: Match query in QA dataset
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
    # STEP 6: Evaluate retrieval outcome
    # -------------------------------
    overlap = set(retrieved_ids) & gold_chunks

    if overlap:
        print("\n✅ Good Result: Retrieved chunks include the gold answer context.")
        print(f"Matched gold chunks: {overlap}")
    else:
        print("\nℹ️ Relative Result: Retrieved chunks did not include the gold answer context.")
        print("But here are the retrieved IDs for inspection:")
        print(retrieved_ids)
