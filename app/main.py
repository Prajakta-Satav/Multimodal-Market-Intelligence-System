import os
import json  # <-- ADD THIS
from fastapi import FastAPI, HTTPException
from app.graph.workflow import workflow
from app.models.schemas import QueryRequest, AnswerResponse
from app.core.logging import logger
from app.core.config import settings
from app.models.state import AgentState

# Services
from app.services.ingestion_pipeline import run_pipeline
from app.services.minio_client import MinioService
from app.services.postgres_client import PostgresService
from app.services.index_builder import (
    build_fundamentals_index,
    build_prices_index,
    build_transcripts_index,
    build_presentations_index,
    build_all_indexes,  # <-- ADD THIS
)
from app.services.embeddings import EmbeddingService
from app.services.chroma_client import ChromaService
from app.services.multi_index_retrieval import multi_index_query


# -------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------

app = FastAPI(title="Multimodal Market Intelligence API")

minio = MinioService()
postgres = PostgresService()
embedder = EmbeddingService()
chroma = ChromaService()

# -------------------------------------------------------------------
# Query endpoint
# -------------------------------------------------------------------

@app.post("/query", response_model=AnswerResponse)
def run_query(req: QueryRequest) -> AnswerResponse:
    # Initialize agent state
    state = AgentState(query=req.query)

    # Run workflow
    final_state = workflow.invoke(state)

    #print("\nIn Main :: \n",final_state, "\n")

    # Build response using your schema
    return AnswerResponse(
        answer=final_state.get("answer",""),
        citations=final_state.get("citations", []),
        evaluation_score=final_state.get("evaluation_score", 0.0),
        confidence=final_state.get("evaluation_score", 0.0),  # reuse score as confidence
        format=final_state.get("output_format", "markdown"),
    )

# -------------------------------------------------------------------
# Ingestion pipeline
# -------------------------------------------------------------------

@app.post("/pipeline/ingest/all")
def ingest_all():
    try:
        result = run_pipeline()
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail="Ingestion failed")

# -------------------------------------------------------------------
# Reset endpoint
# -------------------------------------------------------------------

@app.delete("/reset/all")
def reset_all(confirm: bool = False):
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Confirmation required: set ?confirm=true"
        )
    
    try:
        # --- Clear MinIO buckets ---
        for bucket in [settings.MINIO_BUCKET_DOCS, settings.MINIO_BUCKET_PPT_IMAGES, "audio"]:
            if minio.client.bucket_exists(bucket_name=bucket):
                objects = minio.client.list_objects(bucket_name=bucket, recursive=True)
                for obj in objects:
                    minio.client.remove_object(bucket_name=bucket, object_name=obj.object_name)
                logger.info(f"Cleared bucket {bucket}")
        
        # --- Drop Postgres tables ---
        with postgres.conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS stock_prices CASCADE;")
            cur.execute("DROP TABLE IF EXISTS stock_fundamentals CASCADE;")
            cur.execute("DROP TABLE IF EXISTS transcripts CASCADE;")
            cur.execute("DROP TABLE IF EXISTS balance_sheets CASCADE;")
            cur.execute("DROP TABLE IF EXISTS presentations CASCADE;")
            postgres.conn.commit()
            logger.info("Dropped all Postgres tables")
        
        # --- Recreate schema ---
        postgres._init_schema()
        logger.info("Recreated Postgres schema")

        # --- Reset Chroma collections ---
        try:
            collections = chroma.client.list_collections()
            for col in collections:
                chroma.client.delete_collection(col.name)
                logger.info(f"Deleted Chroma collection: {col.name}")
        except Exception as e:
            logger.warning(f"Chroma reset failed or no collections found: {e}")

        return {
            "status": "success",
            "message": "MinIO buckets cleared, Postgres schema reset, and Chroma collections deleted"
        }
    
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail="Reset failed")

# -------------------------------------------------------------------
# Index builders
# -------------------------------------------------------------------

@app.post("/index/build")
# def build_index():
#     try:
#         f_docs, f_embeds = build_fundamentals_index()
#         p_docs, p_embeds = build_prices_index()
#         t_docs, t_embeds = build_transcripts_index()
#         ppt_docs, ppt_embeds = build_presentations_index()
        
#         return {
#             "fundamentals": {"docs": f_docs, "embeddings": f_embeds},
#             "prices": {"docs": p_docs, "embeddings": p_embeds},
#             "transcripts": {"docs": t_docs, "embeddings": t_embeds},
#             "presentations": {"docs": ppt_docs, "embeddings": ppt_embeds},
#         }
#     except Exception as e:
#         logger.error(f"Index build failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")
@app.post("/index/build")
def build_index():
    try:
        # --------- Load existing Q&A dataset if present ---------
        qa_path = os.path.join(settings.QA_DATASET_DIR, "qa_dataset.json")
        if os.path.exists(qa_path):
            logger.info(f"[Index] Q&A dataset exists at {qa_path}, skipping Q&A generation")
            generate_qa = False
            existing_qa_pairs = []
            with open(qa_path, 'r', encoding='utf-8') as f:
                existing_qa_pairs = json.load(f)
            # prepare a set of questions to prevent duplicates
            for q in existing_qa_pairs:
                if 'question' in q:
                    seen_questions.add(q['question'].lower().strip())
        else:
            generate_qa = True

        # --------- Build all indexes in sequence ----------
        results = build_all_indexes(generate_qa=generate_qa)

        # --------- Collect total docs and Q&A count ----------
        total_docs = sum(r[0] for r in results.values())  # total docs
        total_qa = sum(len(r[2]) for r in results.values())  # total Q&A pairs

        # --------- Save combined Q&A dataset if new or not exists ---------
        if generate_qa:
            all_qa_combined = []
            for r in results.values():
                all_qa_combined.extend(r[2])
            # Remove duplicates across all Q&As
            unique_questions = set()
            final_qa = []
            for qa in all_qa_combined:
                q_text = qa.get('question', '').lower().strip()
                if q_text and q_text not in unique_questions:
                    unique_questions.add(q_text)
                    final_qa.append(qa)
            if final_qa:
                os.makedirs(os.path.dirname(qa_path), exist_ok=True)
                with open(qa_path, 'w', encoding='utf-8') as f:
                    json.dump(final_qa, f, indent=2, ensure_ascii=False)
                logger.info(f"[Index] Saved total {len(final_qa)} Q&A pairs to {qa_path}")

        return {
            "status": "success",
            "total_documents": total_docs,
            "total_embeddings": total_docs,  # assuming same number of docs/embeddings
            "total_qa_pairs": len(final_qa) if generate_qa else 0,
            "qa_dataset_path": qa_path,
            "indexes": {k: {"docs": r[0], "embeddings": r[1], "qa_pairs": len(r[2])} for k, r in results.items()}
        }
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(status_code=500, detail=f"Index build failed: {str(e)}")


# -------------------------------------------------------------------
# Query helpers
# -------------------------------------------------------------------

def rerank_results(res, preferred_order=None, n=10):
    if preferred_order is None:
        preferred_order = ["transcript", "presentation", "fundamentals", "prices"]
    
    docs, metas, ids, dists = res["documents"], res["metadatas"], res["ids"], res["distances"]
    
    items = [
        {"doc": d, "meta": m, "id": i, "dist": dist}
        for dlist, mlist, ilist, distlist in zip(docs, metas, ids, dists)
        for d, m, i, dist in zip(dlist, mlist, ilist, distlist)
    ]
    
    def sort_key(x):
        try:
            type_priority = preferred_order.index(x["meta"]["type"])
        except (ValueError, KeyError):
            type_priority = len(preferred_order)
        return (type_priority, x["dist"])
    
    items.sort(key=sort_key)
    return items[:n]

@app.post("/index/query")
def query_index(q: str, n: int = 10):
    try:
        q_emb = embedder.embed([q])
        res = chroma.query(query_embeddings=q_emb, n_results=n*2)
        reranked = rerank_results(res, n=n)
        
        return {
            "documents": [item["doc"] for item in reranked],
            "metadatas": [item["meta"] for item in reranked],
            "ids": [item["id"] for item in reranked],
            "distances": [item["dist"] for item in reranked],
        }
    except Exception as e:
        logger.error(f"Index query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/index/multi_query")
def multi_query_endpoint(q: str, n_per_index: int = 3, final_n: int = 3):
    try:
        reranked = multi_index_query(q, n_per_index=n_per_index, final_n=final_n)
        
        return {
            "documents": [item["doc"] for item in reranked],
            "metadatas": [item["meta"] for item in reranked],
            "ids": [item["id"] for item in reranked],
            "distances": [item["dist"] for item in reranked],
        }
    except Exception as e:
        logger.error(f"Multi-index query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-query failed: {str(e)}")
