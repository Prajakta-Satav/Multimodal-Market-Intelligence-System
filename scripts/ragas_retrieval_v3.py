# ============================================================================
# Enhanced Retrieval Evaluation (Semantic + IR + LLM RAGAS-style)
# Dataset-wide and single-query modes with logging and robust metrics
# ============================================================================
import os
import re
import json
import csv
import math
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
from app.services.embeddings import EmbeddingService  # your embedding service
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
SIM_THRESHOLDS = [0.70, 0.75, 0.80]  # sweep to calibrate relevance sensitivity
OUTPUT_JSON = "retrieval_eval_results.json"
OUTPUT_CSV = "retrieval_eval_results.csv"

# Gemini setup (expects environment variable GEMINI_API_KEY)
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_KEY = ""
GEMINI_MODEL_NAME = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL_NAME = "gemini-2.5-flash"
    except Exception:
        GEMINI_MODEL_NAME = None

# -------------------------------
# Data containers
# -------------------------------
@dataclass
class QueryEvalResult:
    question: str
    retrieved_ids: List[str]
    gold_ids: List[str]
    overlap_any: bool
    mrr: float
    ndcg_binary: Dict[str, float]        # ndcg@k (binary)
    map_score: float                     # mean average precision (binary)
    semantic_precision: Dict[str, float] # precision@threshold
    semantic_recall: Dict[str, float]    # recall@threshold
    semantic_ndcg: Dict[str, Dict[str, float]]  # ndcg@k for each threshold
    relative_similarity_mean: float
    llm_text: str
    llm_scores: Dict[str, float]         # parsed numeric scores if present
    errors: Optional[str] = None

# -------------------------------
# Utility helpers
# -------------------------------
def ensure_list_strings(xs: Any) -> List[str]:
    if not xs:
        return []
    return [str(x) for x in xs]

def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else 0.0

def batch_embed_texts(embedder: EmbeddingService, texts: List[str]) -> List[List[float]]:
    # If your EmbeddingService supports a batch method, replace this loop with it.
    return [embedder.embed_query(t) for t in texts]

# -------------------------------
# Standard IR metrics (binary)
# -------------------------------
def mean_reciprocal_rank(relevant_ids: set, retrieved_ids: List[str]) -> float:
    for i, cid in enumerate(retrieved_ids):
        if cid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k_binary(relevant_ids: set, retrieved_ids: List[str], k: int) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        rel = 1 if cid in relevant_ids else 0
        dcg += rel / math.log2(i + 2)
    ideal_rels = [1] * min(len(relevant_ids), k)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def average_precision_binary(relevant_ids: set, retrieved_ids: List[str]) -> float:
    # AP = average of precision@k over ranks where a relevant item is found
    num_hits = 0
    precisions = []
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            num_hits += 1
            precisions.append(num_hits / i)
    return safe_mean(precisions)

# -------------------------------
# Semantic metrics
# -------------------------------
def build_similarity_matrix(
    embedder: EmbeddingService,
    retrieved_texts: List[str],
    gold_texts: List[str]
) -> np.ndarray:
    if not retrieved_texts or not gold_texts:
        return np.zeros((len(retrieved_texts), len(gold_texts)))
    retr_emb = batch_embed_texts(embedder, retrieved_texts)
    gold_emb = batch_embed_texts(embedder, gold_texts)
    return cosine_similarity(retr_emb, gold_emb)

def semantic_precision_recall(sim_matrix: np.ndarray, threshold: float) -> Tuple[float, float]:
    if sim_matrix.size == 0:
        return 0.0, 0.0
    # Precision: fraction of retrieved chunks with at least one gold match above threshold
    relevant_retrieved = sum(1 for row in sim_matrix if np.max(row) >= threshold)
    precision = relevant_retrieved / sim_matrix.shape[0] if sim_matrix.shape[0] > 0 else 0.0
    # Recall: fraction of gold chunks covered by at least one retrieved chunk above threshold
    relevant_gold = sum(1 for col in sim_matrix.T if np.max(col) >= threshold)
    recall = relevant_gold / sim_matrix.shape[1] if sim_matrix.shape[1] > 0 else 0.0
    return precision, recall

def semantic_ndcg_at_k(sim_matrix: np.ndarray, k: int) -> float:
    # Graded relevance: for each retrieved chunk, use max similarity to any gold
    if sim_matrix.size == 0:
        return 0.0
    relevances = [float(np.max(sim_matrix[i])) for i in range(min(k, sim_matrix.shape[0]))]
    # DCG
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))
    # IDCG: sort relevances descending
    ideal_rels = sorted(relevances, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0

def relative_similarity_mean(sim_matrix: np.ndarray) -> float:
    return float(np.mean(sim_matrix)) if sim_matrix.size > 0 else 0.0

# -------------------------------
# Chroma helpers
# -------------------------------
def get_gold_chunk_texts(
    collection: chromadb.api.models.Collection.Collection,
    gold_ids: List[str]
) -> List[str]:
    if not gold_ids:
        return []
    fetched = collection.get(ids=gold_ids)
    docs = fetched.get("documents", [])
    return docs or []

# -------------------------------
# LLM-based RAGAS (textual + parsed numeric)
# -------------------------------
LLM_PROMPT = """You are evaluating retrieval quality for a RAG system at retrieval-level.

Query:
{query}

Retrieved Chunks:
{retrieved_fmt}

Gold Chunks:
{gold_fmt}

Rate these metrics (0.0–1.0) with one-sentence reasoning each:
Faithfulness: <score> - <reason>
Answer Relevancy: <score> - <reason>
Context Precision: <score> - <reason>
Context Recall: <score> - <reason>
"""

def format_for_prompt(chunks: List[str], label: str, max_each: int = 500, max_total: int = 4000) -> str:
    lines = []
    total = 0
    for i, t in enumerate(chunks):
        snippet = t.strip().replace("\n", " ")
        if len(snippet) > max_each:
            snippet = snippet[:max_each] + "..."
        entry = f"- {label}#{i+1}: {snippet}"
        if total + len(entry) > max_total:
            lines.append(f"... (truncated {len(chunks) - i} more)")
            break
        lines.append(entry)
        total += len(entry)
    return "\n".join(lines) if lines else "- (none)"

def llm_assess_retrieval(query: str, retrieved_texts: List[str], gold_texts: List[str]) -> str:
    if GEMINI_MODEL_NAME is None:
        return "LLM scoring skipped (GEMINI_API_KEY not set)."
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        prompt = LLM_PROMPT.format(
            query=query,
            retrieved_fmt=format_for_prompt(retrieved_texts, "Retrieved"),
            gold_fmt=format_for_prompt(gold_texts, "Gold")
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        return text.strip() if text.strip() else "LLM returned no text."
    except Exception as e:
        return f"LLM error: {e}"

def parse_llm_scores(text: str) -> Dict[str, float]:
    scores = {}
    patterns = {
        "Faithfulness": r"Faithfulness:\s*([0-9]*\.?[0-9]+)",
        "Answer Relevancy": r"Answer Relevancy:\s*([0-9]*\.?[0-9]+)",
        "Context Precision": r"Context Precision:\s*([0-9]*\.?[0-9]+)",
        "Context Recall": r"Context Recall:\s*([0-9]*\.?[0-9]+)"
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                scores[key] = float(m.group(1))
            except:
                pass
    return scores

# -------------------------------
# Core evaluation per query
# -------------------------------
def evaluate_query(
    collection,
    embedder: EmbeddingService,
    question: str,
    gold_ids: List[str],
    gold_texts_opt: Optional[List[str]] = None
) -> QueryEvalResult:
    errors = None
    retrieved_ids, retrieved_docs = [], []
    gold_texts = gold_texts_opt or []

    try:
        # Retrieve
        q_emb = embedder.embed_query(question)
        results = collection.query(
            query_embeddings=[q_emb],
            n_results=N_RESULTS,
            include=["documents", "ids"]
        )
        retrieved_ids = results["ids"][0]
        retrieved_docs = results["documents"][0]
    except Exception as e:
        errors = f"Retrieval error: {e}\n{traceback.format_exc()}"

    # Ensure gold texts available
    try:
        if not gold_texts:
            gold_texts = get_gold_chunk_texts(collection, gold_ids)
    except Exception as e:
        errors = (errors or "") + f"\nGold fetch error: {e}\n{traceback.format_exc()}"

    relevant_ids = set(gold_ids)
    overlap_any = bool(set(retrieved_ids) & relevant_ids)

    # IR metrics
    mrr = mean_reciprocal_rank(relevant_ids, retrieved_ids)
    ndcg_bin = {f"ndcg@{k}": ndcg_at_k_binary(relevant_ids, retrieved_ids, k) for k in K_VALUES}
    map_score = average_precision_binary(relevant_ids, retrieved_ids)

    # Semantic metrics
    sim_matrix = build_similarity_matrix(embedder, retrieved_docs, gold_texts)
    rel_sim_mean = relative_similarity_mean(sim_matrix)

    sem_precision = {}
    sem_recall = {}
    sem_ndcg = {}
    for th in SIM_THRESHOLDS:
        p, r = semantic_precision_recall(sim_matrix, threshold=th)
        sem_precision[f"precision@{th}"] = p
        sem_recall[f"recall@{th}"] = r
        sem_ndcg[str(th)] = {f"ndcg@{k}": semantic_ndcg_at_k(sim_matrix, k) for k in K_VALUES}

    # LLM assessment
    llm_text = llm_assess_retrieval(question, retrieved_docs, gold_texts)
    llm_scores = parse_llm_scores(llm_text)

    return QueryEvalResult(
        question=question,
        retrieved_ids=retrieved_ids,
        gold_ids=gold_ids,
        overlap_any=overlap_any,
        mrr=mrr,
        ndcg_binary=ndcg_bin,
        map_score=map_score,
        semantic_precision=sem_precision,
        semantic_recall=sem_recall,
        semantic_ndcg=sem_ndcg,
        relative_similarity_mean=rel_sim_mean,
        llm_text=llm_text,
        llm_scores=llm_scores,
        errors=errors
    )

# -------------------------------
# Reporting helpers
# -------------------------------
def print_query_result(res: QueryEvalResult):
    print("\n================ Query Evaluation ================")
    print(f"Question: {res.question}")
    print(f"Overlap with gold: {'Yes' if res.overlap_any else 'No'}")
    print(f"MRR: {res.mrr:.4f}")
    print(f"MAP: {res.map_score:.4f}")

    print("\nBinary NDCG:")
    for k, v in res.ndcg_binary.items():
        print(f"- {k}: {v:.4f}")

    print("\nSemantic Precision (by threshold):")
    for k, v in res.semantic_precision.items():
        print(f"- {k}: {v:.4f}")

    print("\nSemantic Recall (by threshold):")
    for k, v in res.semantic_recall.items():
        print(f"- {k}: {v:.4f}")

    print("\nSemantic NDCG (graded relevance):")
    for th, ndcgs in res.semantic_ndcg.items():
        print(f"- Threshold {th}:")
        for k, v in ndcgs.items():
            print(f"  • {k}: {v:.4f}")

    print(f"\nRelative semantic similarity (mean cosine): {res.relative_similarity_mean:.4f}")

    print("\n🧠 LLM-based RAGAS-style assessment (textual):")
    print(res.llm_text)

    if res.llm_scores:
        print("\nParsed LLM numeric scores:")
        for k, v in res.llm_scores.items():
            print(f"- {k}: {v:.4f}")

    if res.errors:
        print("\nErrors:")
        print(res.errors)

def save_results_json(all_results: List[QueryEvalResult], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)

def save_results_csv(all_results: List[QueryEvalResult], path: str):
    # Flatten key metrics for CSV
    fieldnames = [
        "question", "overlap_any", "mrr", "map_score", "relative_similarity_mean",
        # One representative ndcg@10 (binary) and precision/recall@0.75 for quick comparison
        "ndcg@10_binary", "precision@0.75", "recall@0.75",
        "llm_Faithfulness", "llm_Answer Relevancy", "llm_Context Precision", "llm_Context Recall"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_results:
            row = {
                "question": r.question,
                "overlap_any": r.overlap_any,
                "mrr": f"{r.mrr:.4f}",
                "map_score": f"{r.map_score:.4f}",
                "relative_similarity_mean": f"{r.relative_similarity_mean:.4f}",
                "ndcg@10_binary": f"{r.ndcg_binary.get('ndcg@10', 0.0):.4f}",
                "precision@0.75": f"{r.semantic_precision.get('precision@0.75', 0.0):.4f}",
                "recall@0.75": f"{r.semantic_recall.get('recall@0.75', 0.0):.4f}",
                "llm_Faithfulness": f"{r.llm_scores.get('Faithfulness', 0.0):.4f}" if r.llm_scores else "",
                "llm_Answer Relevancy": f"{r.llm_scores.get('Answer Relevancy', 0.0):.4f}" if r.llm_scores else "",
                "llm_Context Precision": f"{r.llm_scores.get('Context Precision', 0.0):.4f}" if r.llm_scores else "",
                "llm_Context Recall": f"{r.llm_scores.get('Context Recall', 0.0):.4f}" if r.llm_scores else ""
            }
            w.writerow(row)

# -------------------------------
# Main (single or dataset-wide)
# -------------------------------
def main():
    # Connect to Chroma
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    print(client.list_collections())
    collection = client.get_collection(name=COLLECTION_NAME)

    # Load QA dataset
    with open(QA_FILE, "r", encoding="utf-8") as f:
        qa_dataset = json.load(f)
    print(f"✅ Loaded {len(qa_dataset)} queries from {QA_FILE}")

    embedder = EmbeddingService(model_name=EMBEDDING_MODEL_NAME)

    # Mode selection: interactive single query or press Enter to run dataset-wide
    user_query = input("Enter your query (or press Enter to evaluate the whole dataset): ").strip()

    all_results: List[QueryEvalResult] = []

    if user_query:
        matched = next(
            (q for q in qa_dataset if q.get("question", "").strip().lower() == user_query.lower()),
            None
        )
        if not matched:
            print("⚠️ Query not found in QA dataset.")
            return

        gold_ids = ensure_list_strings(matched.get("combined_chunk_ids", []))
        gold_texts_opt = matched.get("combined_chunks_text", [])
        res = evaluate_query(collection, embedder, user_query, gold_ids, gold_texts_opt)
        print_query_result(res)
        all_results.append(res)
    else:
        print("▶ Running dataset-wide evaluation...")
        start = time.time()
        for idx, item in enumerate(qa_dataset, start=1):
            question = item.get("question", "").strip()
            gold_ids = ensure_list_strings(item.get("combined_chunk_ids", []))
            gold_texts_opt = item.get("combined_chunks_text", [])
            if not question:
                continue
            res = evaluate_query(collection, embedder, question, gold_ids, gold_texts_opt)
            print_query_result(res)
            all_results.append(res)
            if idx % 10 == 0:
                print(f"... processed {idx} queries")

        elapsed = time.time() - start
        print(f"\n⏱️ Completed dataset evaluation in {elapsed:.2f}s")

    # Save results
    save_results_json(all_results, OUTPUT_JSON)
    save_results_csv(all_results, OUTPUT_CSV)
    print(f"\n📝 Saved results to: {OUTPUT_JSON} and {OUTPUT_CSV}")

    # Aggregate summary (dataset mode or single)
    if len(all_results) > 1:
        print("\n================ Aggregate Summary ================")
        # Averages across queries
        avg_mrr = safe_mean([r.mrr for r in all_results])
        avg_map = safe_mean([r.map_score for r in all_results])
        avg_ndcg10_bin = safe_mean([r.ndcg_binary.get("ndcg@10", 0.0) for r in all_results])
        avg_prec075 = safe_mean([r.semantic_precision.get("precision@0.75", 0.0) for r in all_results])
        avg_rec075 = safe_mean([r.semantic_recall.get("recall@0.75", 0.0) for r in all_results])
        avg_rel_sim = safe_mean([r.relative_similarity_mean for r in all_results])

        print(f"Avg MRR: {avg_mrr:.4f}")
        print(f"Avg MAP: {avg_map:.4f}")
        print(f"Avg Binary NDCG@10: {avg_ndcg10_bin:.4f}")
        print(f"Avg Semantic Precision@0.75: {avg_prec075:.4f}")
        print(f"Avg Semantic Recall@0.75: {avg_rec075:.4f}")
        print(f"Avg Relative Similarity (mean cosine): {avg_rel_sim:.4f}")

if __name__ == "__main__":
    main()
