# app/services/chroma_client.py

import chromadb
from app.core.logging import logger

class ChromaService:
    def __init__(self, persist_dir: str = "data/chroma"):
        # Persistent client
        self.client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"[ChromaDB] Persistent client initialized at {persist_dir}")
        self.collections = {}

    def get_or_create_collection(self, name: str = "mmi_index", metadata: dict = None):
        """Get or create a collection by name."""
        if name not in self.collections:
            self.collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata=metadata or {"hnsw:space": "cosine"}
            )
            logger.info(f"[ChromaDB] Collection '{name}' ready")
        return self.collections[name]

    def upsert(self, ids, embeddings, metadatas, documents, collection_name: str = "mmi_index"):
        """Upsert documents with embeddings into the collection."""
        collection = self.get_or_create_collection(collection_name)
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents
        )
        logger.info(f"[ChromaDB] Upserted {len(ids)} documents into '{collection_name}'")

    def query(self, query_embeddings, n_results: int = 10, where: dict = None, collection_name: str = "mmi_index"):
        """Query the collection with embeddings."""
        collection = self.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
        logger.info(f"[ChromaDB] Queried '{collection_name}', returned {len(results.get('ids', [[]])[0])} results")
        return results

    def get(self, where: dict = None, limit: int = None, collection_name: str = "mmi_index"):
        """Fetch documents by metadata filter without semantic search."""
        collection = self.get_or_create_collection(collection_name)
        results = collection.get(where=where, limit=limit)
        logger.info(f"[ChromaDB] Fetched {len(results.get('ids', []))} docs from '{collection_name}'")
        return results
