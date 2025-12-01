# app/services/embeddings.py

from typing import List
from sentence_transformers import SentenceTransformer
import tiktoken
from app.core.logging import logger

class EmbeddingService:
    """
    Optimized embedding service for financial data.
    """
    SUPPORTED_MODELS = {
        'finbert': 'ProsusAI/finbert',
        'e5-large': 'intfloat/e5-large-v2',
        'e5-base': 'intfloat/e5-base-v2',
    }
    
    def __init__(self, model_name: str = "e5-large"):
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)
        logger.info(f"[Embeddings] Loading model: {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model_name = model_name.lower()
        self.use_e5_prefix = self.model_name.startswith("e5")
        
        # Fast tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"[Embeddings] Loaded. Dimension: {self.embedding_dim}")
    
    # ----------------------- PREPARATION -----------------------------------------
    
    def _prep_text(self, text: str) -> str:
        """Add E5 prefix if needed."""
        if self.use_e5_prefix and not text.startswith("passage: "):
            return f"passage: {text}"
        return text
    
    def _prep_query(self, query: str) -> str:
        """Add E5 query prefix if needed."""
        if self.use_e5_prefix and not query.startswith("query: "):
            return f"query: {query}"
        return query
    
    # ----------------------- TOKENIZATION -----------------------------------------
    
    def count_tokens(self, text: str) -> int:
        """Fast token counting."""
        return len(self.tokenizer.encode(text))
    
    def chunk_by_tokens(self, text: str, max_tokens: int = 512, overlap_tokens: int = 50) -> List[str]:
        """
        Chunk text by token count with overlap.
        """
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end >= len(tokens):
                break
            start = end - overlap_tokens
        
        logger.info(f"[Embeddings] Chunked text into {len(chunks)} token-based chunks")
        return chunks
    
    def chunk_by_sentences(self, text: str, max_tokens: int = 512, overlap_sentences: int = 2) -> List[str]:
        """
        Chunk text by sentences, respecting token limit.
        """
        # Simple sentence splitter
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                # Keep last N sentences for overlap
                current_chunk = current_chunk[-overlap_sentences:] if len(current_chunk) > overlap_sentences else []
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"[Embeddings] Chunked text into {len(chunks)} sentence-based chunks")
        return chunks
    
    # ----------------------- EMBEDDING -----------------------------------------
    
    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        """
        if is_query:
            texts = [self._prep_query(t) for t in texts]
        else:
            texts = [self._prep_text(t) for t in texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        logger.info(f"[Embeddings] Generated {len(embeddings)} embeddings")
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Convenience method for single query embedding."""
        return self.embed([query], is_query=True)[0]
    
    def embed_document(self, document: str) -> List[float]:
        """Convenience method for single document embedding."""
        return self.embed([document], is_query=False)[0]