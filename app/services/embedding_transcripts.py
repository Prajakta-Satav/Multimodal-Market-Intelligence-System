import os
from datetime import datetime
from typing import List
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb

from app.services.postgresclient import PostgresService  # Adjust your import path if needed

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "earnings_transcripts"


def initialize_clients():
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Earnings call transcripts embeddings"},
    )
    return embedding_model, collection


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def fetch_transcripts_from_postgres(postgres_service, limit=100):
    """
    Fetch transcripts from the table with columns:
    company, quarter, year, text, file_path, uploaded_at
    """
    query = """
        SELECT company, quarter, year, text, file_path, uploaded_at
        FROM transcripts
        ORDER BY company, year, quarter
        LIMIT %s
    """
    rows = postgres_service.query(query, (limit,))
    transcripts = [
        {
            "company": row[0],
            "quarter": row[1],
            "year": row[2],
            "text": row[3],
            "file_path": row[4],
            "uploaded_at": row[5],
        }
        for row in rows
    ]
    return transcripts


def process_all_transcripts_via_postgres(
    postgres_service,
    embedding_model,
    collection,
    sentences_per_chunk=5,
    limit=100,
):
    transcripts = fetch_transcripts_from_postgres(postgres_service, limit=limit)
    print(f"Fetched {len(transcripts)} transcripts from Postgres")

    for t in transcripts:
        source_label = f"{t['company']}_{t['year']}_{t['quarter']}"
        chunks = chunk_by_sentences(t["text"], sentences_per_chunk)
        embeddings = embedding_model.encode(chunks, show_progress_bar=False)

        ids = [f"{source_label}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": source_label,
                "company": t["company"],
                "year": t["year"],
                "quarter": t["quarter"],
                "file_path": t["file_path"],
                "uploaded_at": str(t["uploaded_at"]),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks,
            metadatas=metadatas,
        )
        print(f"Stored {len(chunks)} chunks for {source_label} in ChromaDB")


if __name__ == "__main__":
    postgres_service = PostgresService()
    embedding_model, collection = initialize_clients()
    process_all_transcripts_via_postgres(
        postgres_service,
        embedding_model,
        collection,
        sentences_per_chunk=5,
        limit=100
    )
    print("Embedding from Postgres transcripts complete.")
