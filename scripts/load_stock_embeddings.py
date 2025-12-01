"""
CLI script to load stock embeddings from PostgreSQL to ChromaDB
Fixed import paths for Windows
Usage: python scripts/load_stock_embeddings.py --type both
"""

import argparse
import sys
import os

# Add parent directory to Python path (works on Windows and Linux)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now imports will work
from app.services.stock_embeddings import StockEmbeddingsService
from app.core.logging import logger


def main():
    parser = argparse.ArgumentParser(description="Create stock embeddings from PostgreSQL")
    parser.add_argument('--type', choices=['prices', 'fundamentals', 'both'], default='both',
                        help='Type of data to embed (default: both)')
    parser.add_argument('--ticker', type=str, help='Filter by ticker (e.g., AMZN)')
    parser.add_argument('--batch-size', type=int, default=100, 
                        help='Batch size for processing (default: 100)')
    parser.add_argument('--persist-path', type=str, default='./data/chroma_data',
                        help='ChromaDB persistence path (default: ./data/chroma_data)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("STOCK EMBEDDINGS CREATION")
    print("="*80)
    print(f"Type: {args.type}")
    print(f"Ticker filter: {args.ticker or 'All'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Persist path: {args.persist_path}")
    print("="*80)
    
    # Initialize service
    logger.info("Initializing StockEmbeddingsService...")
    try:
        service = StockEmbeddingsService(chroma_persist_path=args.persist_path)
    except Exception as e:
        print(f"\n❌ Error initializing service: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your database credentials in app/core/config.py")
        print("3. Verify stock_fundamentals table exists and has data")
        return
    
    total_created = 0
    
    # Create price embeddings
    if args.type in ['prices', 'both']:
        print("\n📊 Creating stock price embeddings...")
        try:
            price_count = service.create_price_embeddings(
                ticker=args.ticker, 
                batch_size=args.batch_size
            )
            total_created += price_count
            print(f"✅ Created {price_count} price embeddings")
        except Exception as e:
            print(f"❌ Error creating price embeddings: {e}")
    
    # Create fundamental embeddings
    if args.type in ['fundamentals', 'both']:
        print("\n📈 Creating stock fundamental embeddings...")
        try:
            fund_count = service.create_fundamental_embeddings(
                ticker=args.ticker, 
                batch_size=args.batch_size
            )
            total_created += fund_count
            print(f"✅ Created {fund_count} fundamental embeddings")
        except Exception as e:
            print(f"❌ Error creating fundamental embeddings: {e}")
    
    print("\n" + "="*80)
    print(f"✅ EMBEDDING CREATION COMPLETE - Total: {total_created}")
    print("="*80)
    
    # Verification
    print("\n🔍 Verifying embeddings...")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=args.persist_path)
        collections = client.list_collections()
        
        print(f"\nCollections found: {len(collections)}")
        for coll in collections:
            print(f"  ✓ {coll.name}: {coll.count()} embeddings")
        
        print("\n✅ All embeddings stored successfully!")
        
    except Exception as e:
        print(f"\n⚠️  Verification warning: {e}")


if __name__ == "__main__":
    main()