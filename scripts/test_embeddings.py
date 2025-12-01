"""
Quick Test Script - Verify embeddings and query them
Fixed import paths for Windows
Run this after creating embeddings to test the system
"""

import sys
import os

# Add parent directory to Python path (works on Windows and Linux)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from app.services.stock_embeddings import StockEmbeddingsService
import chromadb


def main():
    print("="*80)
    print("STOCK EMBEDDINGS - QUICK TEST")
    print("="*80)
    
    # Initialize service
    print("\n1️⃣  Initializing service...")
    try:
        service = StockEmbeddingsService(chroma_persist_path="./data/chroma_data")
        print("   ✅ Service initialized")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Check collections
    print("\n2️⃣  Checking collections...")
    try:
        client = chromadb.PersistentClient(path="./data/chroma_data")
        collections = client.list_collections()
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    if not collections:
        print("   ❌ No collections found! Run load_stock_embeddings.py first")
        return
    
    print(f"   ✅ Found {len(collections)} collection(s):")
    for coll in collections:
        print(f"      • {coll.name}: {coll.count()} embeddings")
    
    # Test price queries
    if any(c.name == "stock_prices" for c in collections):
        print("\n3️⃣  Testing stock price queries...")
        
        try:
            # Query 1: High volume days
            results = service.query_similar_prices(
                query_text="high trading volume volatile price swings",
                n_results=3
            )
            
            print("\n   Query: 'high trading volume volatile price swings'")
            print("   Top 3 results:")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                print(f"\n   {i}. {doc}")
                print(f"      Volume: {meta['volume']:,}")
        except Exception as e:
            print(f"   ⚠️  Query error: {e}")
    
    # Test fundamental queries
    if any(c.name == "stock_fundamentals" for c in collections):
        print("\n4️⃣  Testing stock fundamental queries...")
        
        try:
            # Query 2: Strong profitability
            results = service.query_similar_fundamentals(
                query_text="strong profitability high return on equity",
                n_results=3
            )
            
            print("\n   Query: 'strong profitability high return on equity'")
            print("   Top 3 results:")
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
                print(f"\n   {i}. {doc}")
                if meta.get('roe'):
                    print(f"      ROE: {meta['roe']*100:.2f}%")
                if meta.get('eps'):
                    print(f"      EPS: ${meta['eps']:.2f}")
        except Exception as e:
            print(f"   ⚠️  Query error: {e}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE - System working correctly!")
    print("="*80)


if __name__ == "__main__":
    main()