import os
import io
from minio import Minio
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
# -------------------------------------------
# Configuration
# -------------------------------------------
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'admin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'admin123')
TRANSCRIPT_BUCKET =  "transcript-bucket"  # Bucket where transcripts are stored

# Choose your embedding model
# Option 1: General purpose (fast, lightweight)
#EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Option 2: Finance-specific (recommended for earnings calls)
# EMBEDDING_MODEL = "ProsusAI/finbert"
EMBEDDING_MODEL = "intfloat/e5-large-v2"  # For Fin-E5

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "earnings_transcripts"


# -------------------------------------------
# Initialize Clients
# -------------------------------------------
def initialize_clients():
    """Initialize MinIO and ChromaDB clients"""
    
    # MinIO client
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    # Embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # ChromaDB client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Get or create collection
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Earnings call transcripts embeddings"}
    )
    
    return minio_client, embedding_model, collection


# -------------------------------------------
# Text Chunking Functions
# -------------------------------------------
def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start += (chunk_size - overlap)
    
    return chunks


def chunk_by_sentences(text: str, sentences_per_chunk: int = 5) -> List[str]:
    """
    Alternative: Chunk by sentence boundaries (better for semantic coherence)
    
    Args:
        text: Input text
        sentences_per_chunk: Number of sentences per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # Simple sentence splitting (improve with nltk/spacy for production)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ' '.join(sentences[i:i + sentences_per_chunk])
        if chunk.strip():
            chunks.append(chunk.strip())
    
    return chunks


# -------------------------------------------
# Fetch and Process Transcripts
# -------------------------------------------
def list_transcripts(minio_client, bucket: str, prefix: str = "") -> List[str]:
    """List all transcript files in MinIO bucket"""
    try:
        objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
        transcript_files = [obj.object_name for obj in objects if obj.object_name.endswith('.txt')]
        return transcript_files
    except Exception as e:
        print(f"Error listing transcripts: {e}")
        return []


def fetch_transcript(minio_client, bucket: str, object_name: str) -> str:
    """Fetch transcript content from MinIO"""
    try:
        response = minio_client.get_object(bucket, object_name)
        content = response.read().decode('utf-8')
        response.close()
        response.release_conn()
        return content
    except Exception as e:
        print(f"Error fetching {object_name}: {e}")
        return None


def process_transcript(
    transcript_path: str,
    transcript_text: str,
    embedding_model: SentenceTransformer,
    collection,
    chunk_size: int = 512
) -> Dict:
    """
    Process a single transcript: chunk, embed, and store in ChromaDB
    
    Args:
        transcript_path: Path/name of the transcript
        transcript_text: Full transcript text
        embedding_model: SentenceTransformer model
        collection: ChromaDB collection
        chunk_size: Size of text chunks
        
    Returns:
        Processing statistics
    """
    print(f"\nProcessing: {transcript_path}")
    
    # Extract metadata from filename (customize based on your naming convention)
    # Example: "Amazon_2025_Q1.txt" -> company="Amazon", year="2025", quarter="Q1"
    filename = os.path.basename(transcript_path)
    name_parts = filename.replace('.txt', '').split('_')
    
    company = name_parts[0] if len(name_parts) > 0 else "Unknown"
    year = name_parts[1] if len(name_parts) > 1 else "Unknown"
    quarter = name_parts[2] if len(name_parts) > 2 else "Unknown"
    
    # Chunk the text
    chunks = chunk_text(transcript_text, chunk_size=chunk_size, overlap=50)
    print(f"  Created {len(chunks)} chunks")
    
    # Generate embeddings
    print(f"  Generating embeddings...")
    embeddings = embedding_model.encode(chunks, show_progress_bar=False)
    
    # Prepare data for ChromaDB
    ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
    
    metadatas = [
        {
            "source": transcript_path,
            "company": company,
            "year": year,
            "quarter": quarter,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "timestamp": datetime.now().isoformat()
        }
        for i in range(len(chunks))
    ]
    
    # Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas
    )
    
    print(f"  ✓ Stored {len(chunks)} chunks in ChromaDB")
    
    return {
        "transcript": transcript_path,
        "chunks": len(chunks),
        "embedding_dim": embeddings.shape[1]
    }


# -------------------------------------------
# Main Pipeline
# -------------------------------------------
def process_all_transcripts(
    minio_client,
    embedding_model,
    collection,
    bucket: str = TRANSCRIPT_BUCKET,
    prefix: str = "",
    chunk_size: int = 512
):
    """
    Main pipeline: Fetch all transcripts, generate embeddings, store in ChromaDB
    """
    # List all transcripts
    transcript_files = list_transcripts(minio_client, bucket, prefix)
    print(f"Found {len(transcript_files)} transcript files in bucket '{bucket}'")
    
    if not transcript_files:
        print("No transcript files found. Make sure your bucket contains .txt files.")
        return
    
    results = []
    
    for transcript_path in transcript_files:
        # Fetch transcript
        transcript_text = fetch_transcript(minio_client, bucket, transcript_path)
        
        if transcript_text:
            # Process and store
            result = process_transcript(
                transcript_path,
                transcript_text,
                embedding_model,
                collection,
                chunk_size=chunk_size
            )
            results.append(result)
        else:
            print(f"  ✗ Skipped {transcript_path} (fetch failed)")
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total transcripts processed: {len(results)}")
    print(f"Total chunks stored: {sum(r['chunks'] for r in results)}")
    print(f"ChromaDB collection: {COLLECTION_NAME}")
    print(f"Persist directory: {CHROMA_PERSIST_DIR}")
    
    return results


# -------------------------------------------
# Query Functions (Bonus)
# -------------------------------------------
def search_transcripts(collection, query: str, n_results: int = 5, filter_dict: Dict = None):
    """
    Search transcripts using semantic similarity
    
    Args:
        collection: ChromaDB collection
        query: Search query text
        n_results: Number of results to return
        filter_dict: Optional metadata filter (e.g., {"company": "Amazon"})
        
    Returns:
        Search results
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter_dict if filter_dict else None
    )
    
    return results


def example_search(collection):
    """Example searches"""
    print("\n" + "="*60)
    print("EXAMPLE SEARCHES")
    print("="*60)
    
    # Example 1: General search
    print("\n1. Search: 'revenue growth and profitability'")
    results = search_transcripts(collection, "revenue growth and profitability", n_results=3)
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n  Result {i+1} ({metadata['source']} - Chunk {metadata['chunk_index']}):")
        print(f"  {doc[:200]}...")
    
    # Example 2: Filtered search
    print("\n2. Search with filter: 'cloud services' from Amazon")
    results = search_transcripts(
        collection,
        "cloud services and AWS performance",
        n_results=2,
        filter_dict={"company": "Amazon"}
    )
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"\n  Result {i+1} ({metadata['source']}):")
        print(f"  {doc[:200]}...")


# -------------------------------------------
# Main Execution
# -------------------------------------------
if __name__ == "__main__":
    # Initialize
    print("Initializing clients...")
    minio_client, embedding_model, collection = initialize_clients()
    
    # Process all transcripts
    results = process_all_transcripts(
        minio_client,
        embedding_model,
        collection,
        bucket=TRANSCRIPT_BUCKET,
        prefix="",  
        chunk_size=512
    )
    
  
    print("\nDone! Your transcripts are now embedded and stored in ChromaDB.")
    print(f"Collection '{COLLECTION_NAME}' is persisted at: {CHROMA_PERSIST_DIR}")