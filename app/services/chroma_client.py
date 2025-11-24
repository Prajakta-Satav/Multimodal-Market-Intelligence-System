# app/services/chroma_client.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any
from app.core.config import settings

class ChromaService:
    def __init__(self, collection_name: str = "financial_multimodal"):
        self.client = chromadb.Client(
            ChromaSettings(persist_directory=settings.CHROMA_PERSIST_DIR)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def query(self, text: str, n_results: int = 8, where: Dict[str, Any] = None):
        return self.collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where or {},
            include=["documents", "metadatas", "distances", "ids"]
        )
